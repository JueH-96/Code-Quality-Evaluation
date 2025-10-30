def main():
    S = input().strip()
    vowels = set("aeiou")
    result = []
    for ch in S:
        if ch not in vowels:
            result.append(ch)
    print("".join(result))

if __name__ == "__main__":
    main()