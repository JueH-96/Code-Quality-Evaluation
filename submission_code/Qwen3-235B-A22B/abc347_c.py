import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    A = int(input[idx]); idx += 1
    B = int(input[idx]); idx += 1
    C = A + B
    D = list(map(int, input[idx:idx+N]))
    
    if N == 1:
        print("Yes")
        return
    
    m = [d % C for d in D]
    deltas = []
    m0 = m[0]
    for i in range(1, N):
        delta = (m[i] - m0) % C
        deltas.append(delta)
    
    forbidden = []
    for delta in deltas:
        lower = A - delta
        start = max(0, lower)
        upper = C - delta
        end = min(A, upper)
        if start < end:
            forbidden.append((start, end))
    
    # Merge intervals
    forbidden.sort()
    merged = []
    for s, e in forbidden:
        if not merged:
            merged.append([s, e])
        else:
            last_s, last_e = merged[-1]
            if s <= last_e:
                # Overlap or adjacent, merge
                merged[-1][1] = max(e, last_e)
            else:
                merged.append([s, e])
    
    # Check coverage of [0, A)
    covered = 0
    for s, e in merged:
        if s > covered:
            print("Yes")
            return
        covered = max(covered, e)
    if covered < A:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()