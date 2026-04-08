import os
import pdb
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F

try:
    import ctcdecode
    _CTCDECODE_IMPORT_ERROR = None
except Exception as _err:
    ctcdecode = None
    _CTCDECODE_IMPORT_ERROR = _err


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0, beam_width=10, num_processes=10):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = str(search_mode).lower().strip()
        self.blank_id = blank_id
        self.ctc_decoder = None
        self.beam_width = int(beam_width)
        self.num_processes = int(num_processes)
        if self.search_mode != "max":
            if ctcdecode is None:
                print(
                    "[Decode] ctcdecode import failed ({}). "
                    "Fallback to greedy max decode.".format(_CTCDECODE_IMPORT_ERROR)
                )
                self.search_mode = "max"
            else:
                vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
                self.ctc_decoder = ctcdecode.CTCBeamDecoder(
                    vocab,
                    beam_width=self.beam_width,
                    blank_id=blank_id,
                    num_processes=self.num_processes,
                )
                print(
                    "[Decode] ctcdecode import ok. search_mode=beam, beam_width={}, num_processes={}".format(
                        self.beam_width, self.num_processes
                    )
                )
        else:
            print("[Decode] search_mode=max (greedy argmax decode).")

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max" or self.ctc_decoder is None:
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list
