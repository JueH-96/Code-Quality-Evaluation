import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    N = int(input().strip())

    # For each flavor, track the two largest deliciousness values
    # best1[f] = largest S for flavor f
    # best2[f] = second largest S for flavor f
    best1 = [0] * (N + 1)
    best2 = [0] * (N + 1)

    for _ in range(N):
        f, s = map(int, input().split())
        # place s into best1[f] and best2[f] appropriately
        if s > best1[f]:
            best2[f] = best1[f]
            best1[f] = s
        elif s > best2[f]:
            best2[f] = s

    # Compute maximum satisfaction from same-flavor pair
    # which is best1[f] + best2[f]//2 for each flavor f with two cups
    max_same = 0
    for f in range(1, N+1):
        if best2[f] > 0:
            val = best1[f] + best2[f] // 2
            if val > max_same:
                max_same = val

    # Compute maximum satisfaction from different-flavor pair
    # We only need the largest S per flavor; then pick the top two among those.
    tops = [best1[f] for f in range(1, N+1) if best1[f] > 0]
    tops.sort(reverse=True)
    max_diff = 0
    if len(tops) >= 2:
        max_diff = tops[0] + tops[1]

    # The answer is the maximum of the two cases
    ans = max(max_same, max_diff)
    print(ans)

if __name__ == "__main__":
    main()