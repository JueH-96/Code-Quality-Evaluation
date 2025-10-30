import sys

def main():
    import sys
    data = sys.stdin.read().split()
    S = data[0]
    T = data[1]
    
    # helper: is P a subsequence of S?
    def is_subseq(P, S):
        j = 0
        m = len(P)
        for c in S:
            if j < m and c.upper() == P[j]:
                j += 1
                if j == m:
                    return True
        return False
    
    # 1) check if T can be formed by taking 3-letter subsequence
    if is_subseq(T, S):
        print("Yes")
        return
    
    # 2) check if T can be formed by taking 2-letter subsequence + 'X'
    #    that means T[2] must be 'X', and T[0:2] a subsequence of S
    if T[2] == 'X' and is_subseq(T[:2], S):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()