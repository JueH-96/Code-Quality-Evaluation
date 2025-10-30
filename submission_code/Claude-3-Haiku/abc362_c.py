from typing import List

def find_sequence(n: int, ranges: List[List[int]]) -> List[int]:
    total_sum = sum(r[0] for r in ranges)
    if total_sum != 0:
        return ["No"]

    sequence = []
    for l, r in ranges:
        x = l
        if total_sum - l >= 0 and total_sum - l <= r - l:
            x = total_sum - l
        sequence.append(x)
        total_sum -= x

    return ["Yes"] + [str(x) for x in sequence]

if __name__ == "__main__":
    n = int(input())
    ranges = []
    for _ in range(n):
        l, r = map(int, input().split())
        ranges.append([l, r])
    print("
".join(find_sequence(n, ranges)))