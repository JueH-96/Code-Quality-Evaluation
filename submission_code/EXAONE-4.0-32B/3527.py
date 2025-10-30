import sys
import bisect
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    q = int(data[1])
    colors = list(map(int, data[2:2+n]))
    queries = []
    index = 2 + n
    for i in range(q):
        t = int(data[index]); index += 1
        if t == 1:
            L = int(data[index]); index += 1
            queries.append((t, L))
        else:
            idx = int(data[index]); index += 1
            col = int(data[index]); index += 1
            queries.append((t, idx, col))
    
    a = colors[:]
    cuts = set()
    for i in range(n):
        if a[i] == a[(i+1) % n]:
            cuts.add(i)
    sorted_cuts = sorted(cuts)
    freq = defaultdict(int)
    
    if not cuts:
        freq[n] = 1
    else:
        k = len(sorted_cuts)
        for i in range(k):
            l_val = sorted_cuts[i]
            r_val = sorted_cuts[(i+1) % k]
            if r_val >= l_val:
                length_seg = r_val - l_val
            else:
                length_seg = r_val + n - l_val
            freq[length_seg] += 1

    results = []
    for query in queries:
        if query[0] == 1:
            L = query[1]
            if not cuts:
                results.append(str(n))
            else:
                total = 0
                for length_seg, cnt in freq.items():
                    if length_seg >= L:
                        total += (length_seg - L + 1) * cnt
                results.append(str(total))
        else:
            idx = query[1]
            new_color = query[2]
            affected = [(idx - 1 + n) % n, idx]
            
            for cut_pos in affected:
                if cut_pos in cuts:
                    pos = bisect.bisect_left(sorted_cuts, cut_pos)
                    k_now = len(sorted_cuts)
                    if k_now == 1:
                        predecessor = sorted_cuts[0]
                        successor = sorted_cuts[0]
                    else:
                        if pos == 0:
                            predecessor = sorted_cuts[-1]
                        else:
                            predecessor = sorted_cuts[pos-1]
                        if pos == k_now - 1:
                            successor = sorted_cuts[0]
                        else:
                            successor = sorted_cuts[pos+1]
                    
                    if successor >= predecessor:
                        len1 = successor - predecessor
                    else:
                        len1 = successor + n - predecessor
                    freq[len1] = freq.get(len1, 0) - 1
                    if freq[len1] == 0:
                        del freq[len1]
                    
                    cuts.remove(cut_pos)
                    del sorted_cuts[pos]
                    
                    if not cuts:
                        freq[n] = freq.get(n, 0) + 1
                    else:
                        if successor >= predecessor:
                            new_len = successor - predecessor
                        else:
                            new_len = successor + n - predecessor
                        freq[new_len] = freq.get(new_len, 0) + 1
            
            a[idx] = new_color
            
            for cut_pos in affected:
                if a[cut_pos] == a[(cut_pos+1) % n]:
                    if cut_pos not in cuts:
                        if not cuts:
                            if n in freq:
                                freq[n] -= 1
                                if freq[n] == 0:
                                    del freq[n]
                            len1 = cut_pos - 0
                            len2 = (0 + n - cut_pos)
                            freq[len1] = freq.get(len1, 0) + 1
                            freq[len2] = freq.get(len2, 0) + 1
                            cuts.add(cut_pos)
                            bisect.insort(sorted_cuts, cut_pos)
                        else:
                            pos = bisect.bisect_left(sorted_cuts, cut_pos)
                            k_now = len(sorted_cuts)
                            if k_now == 0:
                                predecessor = None
                                successor = None
                            else:
                                if pos == 0:
                                    predecessor = sorted_cuts[-1]
                                else:
                                    predecessor = sorted_cuts[pos-1]
                                if pos < k_now:
                                    successor = sorted_cuts[pos]
                                else:
                                    successor = sorted_cuts[0]
                            
                            if successor >= predecessor:
                                old_len = successor - predecessor
                            else:
                                old_len = successor + n - predecessor
                            freq[old_len] = freq.get(old_len, 0) - 1
                            if freq[old_len] == 0:
                                del freq[old_len]
                            
                            if cut_pos >= predecessor:
                                len1 = cut_pos - predecessor
                            else:
                                len1 = cut_pos + n - predecessor
                            if successor >= cut_pos:
                                len2 = successor - cut_pos
                            else:
                                len2 = successor + n - cut_pos
                            freq[len1] = freq.get(len1, 0) + 1
                            freq[len2] = freq.get(len2, 0) + 1
                            
                            cuts.add(cut_pos)
                            bisect.insort(sorted_cuts, cut_pos)
    
    print("
".join(results))

if __name__ == "__main__":
    main()