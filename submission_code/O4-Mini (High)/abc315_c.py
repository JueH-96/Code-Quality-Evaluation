import sys
input = sys.stdin.readline

def main():
    N = int(input())
    # best1[f]: largest S for flavor f
    # best2[f]: second largest S for flavor f
    best1 = [0] * (N + 1)
    best2 = [0] * (N + 1)

    for _ in range(N):
        f, s = map(int, input().split())
        if s > best1[f]:
            best2[f] = best1[f]
            best1[f] = s
        elif s > best2[f]:
            best2[f] = s

    largest = 0
    second_largest = 0
    same_max = 0

    # Scan through flavors 1..N
    for f in range(1, N + 1):
        s1 = best1[f]
        s2 = best2[f]
        # Update the two largest per-flavor-max values
        if s1 > largest:
            second_largest = largest
            largest = s1
        elif s1 > second_largest:
            second_largest = s1
        # If there are at least two items of this flavor, consider same‐flavor score
        if s2 > 0:
            tmp = s1 + (s2 // 2)
            if tmp > same_max:
                same_max = tmp

    # Best different-flavor satisfaction
    diff_max = largest + second_largest

    # Answer is the max of same‐flavor and different‐flavor cases
    ans = diff_max if diff_max > same_max else same_max
    print(ans)

main()