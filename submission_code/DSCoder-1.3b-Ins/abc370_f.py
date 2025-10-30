import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the total mass of each piece
    total_mass = [0] * (N + 1)
    for i in range(N):
        total_mass[i + 1] = total_mass[i] + A[i]

    # Calculate the cumulative sum of the total mass
    cum_sum = [0] * (N + 1)
    for i in range(N):
        cum_sum[i + 1] = cum_sum[i] + total_mass[i + 1]

    # Calculate the minimum total mass among the K people
    min_total_mass = float('inf')
    for i in range(K):
        min_total_mass = min(min_total_mass, cum_sum[i + 1] - cum_sum[i])

    # Calculate the number of cut lines that are never cut
    never_cut_lines = 0
    for i in range(1, N + 1):
        if total_mass[i] != total_mass[i - 1]:
            never_cut_lines += 1

    print(min_total_mass, never_cut_lines)

if __name__ == "__main__":
    solve()