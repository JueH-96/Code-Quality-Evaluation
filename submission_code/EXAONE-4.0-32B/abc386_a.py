from collections import Counter

def main():
    data = input().split()
    nums = list(map(int, data))
    ctr = Counter(nums)
    freqs = sorted(ctr.values())
    if freqs in [[1, 3], [2, 2]]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()