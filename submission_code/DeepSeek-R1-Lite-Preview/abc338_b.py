from collections import Counter

def main():
    S = input().strip()
    count = Counter(S)
    max_freq = max(count.values())
    candidates = [char for char, freq in count.items() if freq == max_freq]
    candidates_sorted = sorted(candidates)
    print(candidates_sorted[0])

if __name__ == "__main__":
    main()