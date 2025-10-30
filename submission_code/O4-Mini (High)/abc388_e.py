import sys

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    a = list(map(int, data[1:]))

    # We'll try to match the smallest n//2 mochi as tops with the largest as bottoms greedily.
    mid = n // 2
    left = 0
    right = mid
    matches = 0

    # Greedy two-pointer: for each small mochi at `left`, find the earliest big mochi at `right`
    # such that a[left] * 2 <= a[right]. Count the match and advance both pointers.
    while left < mid and right < n:
        if a[left] * 2 <= a[right]:
            matches += 1
            left += 1
            right += 1
        else:
            right += 1

    print(matches)

main()