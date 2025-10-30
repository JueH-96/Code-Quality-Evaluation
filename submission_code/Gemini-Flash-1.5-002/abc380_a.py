n_str = input()
n = int(n_str)

if not 100000 <= n <= 999999:
    print("No")
else:
    counts = {}
    for digit in n_str:
        digit = int(digit)
        counts[digit] = counts.get(digit, 0) + 1

    if counts.get(1, 0) == 1 and counts.get(2, 0) == 2 and counts.get(3, 0) == 3:
        
        valid = True
        for digit in counts:
            if digit not in [1,2,3]:
                valid = False
                break
        if valid:
            print("Yes")
        else:
            print("No")

    else:
        print("No")