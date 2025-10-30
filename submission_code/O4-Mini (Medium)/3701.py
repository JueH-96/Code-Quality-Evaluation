class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        # If string shorter than 3, impossible
        if n < 3:
            return ""
        # Convert caption chars to int 0..25
        a = [ord(c) - 97 for c in caption]
        INF = 10**18

        # prev_comb[pos] is a bytearray of length 26*4 storing for each (ch, len_idx)
        # the encoded prev pointer: prev_ch_enc*4 + prev_len_idx
        # len_idx = len-1 for current state
        prev_comb = [bytearray(26 * 4) for _ in range(n + 1)]

        # dp_prev[ch][len_idx]: minimal cost up to previous pos ending with run of char ch and run-length = len_idx+1 or >=3 if len_idx==2
        dp_prev = [[INF] * 3 for _ in range(26)]

        # best_prev: minimal dp_prev[ch][2] over ch, for runs >=3
        # We keep best and second best (cost,ch) to support new-block transition
        best_cost = 0     # at pos=0 we allow starting any block with cost 0
        best_ch = 26      # special marker for "start" (encoded as 26)
        second_cost = INF
        second_ch = 26

        # Process positions 1..n
        for pos in range(1, n + 1):
            ci = a[pos - 1]
            dp_cur = [[INF] * 3 for _ in range(26)]
            # Build dp_cur
            for ch2 in range(26):
                # cost to change caption[pos-1] to ch2
                cost_i = abs(ci - ch2)
                # 1) new block of ch2, if previous run was >=3
                #    take best_prev (cost,best_ch) 
                if best_cost < INF:
                    tot = best_cost + cost_i
                    # dp_cur[ch2][0] = run-length 1
                    if tot < dp_cur[ch2][0]:
                        dp_cur[ch2][0] = tot
                        # store prev pointer: prev_ch_enc=best_ch, prev_len_idx=2 (len=3)
                        prev_comb[pos][ch2 * 4 + 0] = best_ch * 4 + 2
                # 2) continuation: len1->len2
                c01 = dp_prev[ch2][0]
                if c01 < INF:
                    tot = c01 + cost_i
                    if tot < dp_cur[ch2][1]:
                        dp_cur[ch2][1] = tot
                        # prev from (ch2,len_idx=0)
                        prev_comb[pos][ch2 * 4 + 1] = ch2 * 4 + 0
                # 3) continuation: len2->len3
                c12 = dp_prev[ch2][1]
                if c12 < INF:
                    tot = c12 + cost_i
                    if tot < dp_cur[ch2][2]:
                        dp_cur[ch2][2] = tot
                        # prev from (ch2,len_idx=1)
                        prev_comb[pos][ch2 * 4 + 2] = ch2 * 4 + 1
                # 4) continuation: len3->len3
                c22 = dp_prev[ch2][2]
                if c22 < INF:
                    tot = c22 + cost_i
                    # only update if strictly better, to keep earlier pointer priority
                    if tot < dp_cur[ch2][2]:
                        dp_cur[ch2][2] = tot
                        # prev from (ch2,len_idx=2)
                        prev_comb[pos][ch2 * 4 + 2] = ch2 * 4 + 2

            # Compute best and second best over dp_cur[ch][2]
            nb_cost, nb_ch = INF, 26
            ns_cost, ns_ch = INF, 26
            for ch in range(26):
                c3 = dp_cur[ch][2]
                if c3 < nb_cost or (c3 == nb_cost and ch < nb_ch):
                    # shift best to second
                    ns_cost, ns_ch = nb_cost, nb_ch
                    nb_cost, nb_ch = c3, ch
                elif c3 < ns_cost or (c3 == ns_cost and ch < ns_ch):
                    ns_cost, ns_ch = c3, ch
            # Move current to prev
            dp_prev = dp_cur
            best_cost, best_ch = nb_cost, nb_ch
            second_cost, second_ch = ns_cost, ns_ch

        # At end, pick minimal over dp_prev[ch][2]
        final_cost = INF
        final_ch = -1
        for ch in range(26):
            c3 = dp_prev[ch][2]
            if c3 < final_cost or (c3 == final_cost and ch < final_ch):
                final_cost, final_ch = c3, ch

        if final_cost >= INF:
            return ""

        # Reconstruct by backtracking
        res = [''] * n
        ch = final_ch
        len_idx = 2  # len_idx=2 means run-length>=3
        pos = n
        while pos > 0:
            # set this character
            res[pos - 1] = chr(ch + 97)
            # get prev pointer
            code = prev_comb[pos][ch * 4 + len_idx]
            prev_len_idx = code & 3  # code % 4
            prev_ch_enc = code >> 2   # code // 4
            # If special start marker, we're done
            if prev_ch_enc == 26:
                break
            # move back
            ch = prev_ch_enc
            len_idx = prev_len_idx
            pos -= 1

        return "".join(res)