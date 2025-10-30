import sys
import bisect

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        K = int(input[ptr+1])
        ptr +=2
        A = list(map(int, input[ptr:ptr+N]))
        ptr +=N
        B = list(map(int, input[ptr:ptr+N]))
        ptr +=N
        
        a_set = set(A)
        for b in B:
            if b not in a_set:
                print("No")
                break
        else:
            threshold = N // 2
            if K >= threshold:
                print("Yes")
                continue
            else:
                pos = {}
                for idx, num in enumerate(A):
                    if num not in pos:
                        pos[num] = []
                    pos[num].append(idx)
                possible = True
                for i in range(N):
                    x = B[i]
                    positions = pos[x]
                    low = max(0, i - K)
                    high = min(N-1, i + K)
                    idx = bisect.bisect_left(positions, low)
                    if idx < len(positions):
                        if positions[idx] <= high:
                            continue
                    possible = False
                    break
                if possible:
                    print("Yes")
                else:
                    print("No")
        continue
    return

if __name__ == '__main__':
    main()