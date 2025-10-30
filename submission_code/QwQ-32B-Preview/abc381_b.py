S = input().strip()

if len(S) % 2 != 0:
    print("No")
else:
    # Check if each pair of characters is equal
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            print("No")
            break
    else:
        from collections import Counter
        count = Counter(S)
        # Check if each character appears exactly twice
        for char in count:
            if count[char] != 2:
                print("No")
                break
        else:
            print("Yes")