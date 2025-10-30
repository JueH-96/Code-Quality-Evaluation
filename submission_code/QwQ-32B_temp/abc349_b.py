from collections import Counter

def main():
    S = input().strip()
    char_counts = Counter(S)
    freq_count = {}
    for count in char_counts.values():
        if count in freq_count:
            freq_count[count] += 1
        else:
            freq_count[count] = 1
    max_count = max(char_counts.values()) if char_counts else 0
    
    for i in range(1, max_count + 1):
        current = freq_count.get(i, 0)
        if current not in (0, 2):
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()