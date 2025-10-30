import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    
    # Every single element forms an arithmetic progression by itself
    ans = N
    
    if N >= 2:
        prev_diff = None      # previous difference value in the run
        run_len   = 0         # current run length in the difference array
        
        for i in range(N - 1):
            diff = A[i + 1] - A[i]
            if diff == prev_diff:
                run_len += 1
            else:
                if prev_diff is not None:              # close previous run
                    ans += run_len * (run_len + 1) // 2
                prev_diff = diff
                run_len   = 1
        
        # add the contribution of the final run
        ans += run_len * (run_len + 1) // 2
    
    print(ans)

if __name__ == "__main__":
    main()