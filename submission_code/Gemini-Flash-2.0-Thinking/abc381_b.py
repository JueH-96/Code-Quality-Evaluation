from collections import Counter

s = input()
n = len(s)

if n % 2 != 0:
    print("No")
else:
    # Check condition 2: paired characters
    is_paired = True
    for i in range(0, n, 2):
        if s[i] != s[i+1]:
            is_paired = False
            break

    if not is_paired:
        print("No")
    else:
        # Check condition 3: each character appears exactly twice
        counts = Counter(s)
        all_counts_are_two = True
        for count in counts.values():
            if count != 2:
                all_counts_are_two = False
                break

        if all_counts_are_two:
            print("Yes")
        else:
            print("No")