import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1 + n]))
    
    last_pos = {}
    best = n + 1            # larger than any possible subarray length
    
    for i, val in enumerate(A):
        if val in last_pos:
            # length of the subarray that starts at previous occurrence
            # and ends at current position (both inclusive)
            best = min(best, i - last_pos[val] + 1)
        # update last seen position of val
        last_pos[val] = i
    
    print(best if best <= n else -1)

if __name__ == "__main__":
    main()