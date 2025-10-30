import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    counts = [0] * (n + 1)
    current_max = 0
    current_leader = 0
    for num in a:
        counts[num] += 1
        new_count = counts[num]
        if new_count > current_max:
            current_max = new_count
            current_leader = num
        elif new_count == current_max:
            if num < current_leader:
                current_leader = num
        print(current_leader)

if __name__ == "__main__":
    main()