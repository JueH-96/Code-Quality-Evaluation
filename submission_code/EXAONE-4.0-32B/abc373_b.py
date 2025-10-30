def main():
    s = input().strip()
    target = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    total = 0
    current_index = s.index('A')
    for letter in target[1:]:
        next_index = s.index(letter)
        total += abs(next_index - current_index)
        current_index = next_index
    print(total)

if __name__ == "__main__":
    main()