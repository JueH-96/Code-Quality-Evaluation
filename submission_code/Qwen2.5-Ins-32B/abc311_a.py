def find_min_length(s):
    char_count = {'A': 0, 'B': 0, 'C': 0}
    for i, char in enumerate(s):
        char_count[char] += 1
        if all(value > 0 for value in char_count.values()):
            return i + 1

if __name__ == "__main__":
    N = int(input())
    S = input().strip()
    print(find_min_length(S))