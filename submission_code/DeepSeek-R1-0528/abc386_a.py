from collections import Counter

def main():
    freqs = sorted(Counter(map(int, input().split())).values())
    print("Yes" if freqs in [[1, 3], [2, 2]] else "No")

if __name__ == "__main__":
    main()