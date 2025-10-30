# YOUR CODE HERE
import sys

import sys

import threading

def main():
    import sys
    input = sys.stdin.readline
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    Q = int(sys.stdin.readline())
    m = [i for i in range(26)]  # mapping of letters, indices 0-25
    for _ in range(Q):
        c_i, d_i = sys.stdin.readline().strip().split()
        m_c = ord(c_i) - ord('a')
        m_d = ord(d_i) - ord('a')
        for idx in range(26):
            if m[idx] == m_c:
                m[idx] = m_d
    res = []
    for ch in S:
        idx = ord(ch) - ord('a')
        mapped_idx = m[idx]
        res.append(chr(mapped_idx + ord('a')))
    print(''.join(res))

threading.Thread(target=main).start()