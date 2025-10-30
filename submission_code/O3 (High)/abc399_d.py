import sys

def solve_one(N, A):
    # 1-based copy of the sequence with two sentinels
    B = [0] + A + [0]          #  B[1 .. 2N],  B[2N+1] = 0
    
    first = [0]*(N+1)          # first position of each label
    second = [0]*(N+1)         # second position
    
    for idx, v in enumerate(A, 1):
        if first[v] == 0:
            first[v] = idx
        else:
            second[v] = idx
    
    ans = 0
    limit = 2*N
    for x in range(1, N+1):
        L = first[x]
        R = second[x]
        # each couple must not be adjacent in the original line
        if R == L + 1:
            continue
        # need a seat to the right of the second occurrence
        if R == limit:
            continue
        
        y = B[L+1]             # neighbour just after the first x
        if y == x:             # same label - cannot be a partner
            continue
        if B[R+1] != y:        # the neighbour after the second x must be the same y
            continue
        
        # y must occur exactly at the two seats we expect
        if first[y] == L+1 and second[y] == R+1:
            # y is also not originally adjacent because R-L >= 2
            ans += 1           # each unordered pair is counted only here (with x the 'outer' label)
    return ans


def main() -> None:
    it = iter(sys.stdin.buffer.read().split())
    T  = int(next(it))
    
    out_lines = []
    for _ in range(T):
        N = int(next(it))
        A = [int(next(it)) for _ in range(2*N)]
        out_lines.append(str(solve_one(N, A)))
    
    sys.stdout.write("
".join(out_lines))


if __name__ == "__main__":
    main()