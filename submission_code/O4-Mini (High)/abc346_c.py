import sys

def main():
    input = sys.stdin.readline
    # Read N and K
    line = input().split()
    if not line:
        return
    N, K = map(int, line)
    # Read the sequence A
    A = map(int, input().split())
    # Collect distinct values of A_i that lie in [1, K]
    seen = set()
    for a in A:
        if 1 <= a <= K:
            seen.add(a)
    # Sum of all integers from 1 to K
    total_all = K * (K + 1) // 2
    # Subtract the sum of those that appear in A
    ans = total_all - sum(seen)
    print(ans)

main()