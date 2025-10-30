W, B = map(int, input().split())

if B == 0:
    if W == 0:
        print("Yes")
    else:
        runs_of_w = []
        current_run = 0
        for c in "wbwbwwbwbwbw":
            if c == 'w':
                current_run += 1
            else:
                if current_run > 0:
                    runs_of_w.append(current_run)
                    current_run = 0
        if current_run > 0:
            runs_of_w.append(current_run)
        if W == 1:
            print("Yes")
        else:
            print("No")
elif W == 0:
    if B == 0:
        print("Yes")
    else:
        runs_of_b = []
        current_run = 0
        for c in "wbwbwwbwbwbw":
            if c == 'b':
                current_run += 1
            else:
                if current_run > 0:
                    runs_of_b.append(current_run)
                    current_run = 0
        if current_run > 0:
            runs_of_b.append(current_run)
        if B == 1:
            print("Yes")
        else:
            print("No")
else:
    base = "wbwbwwbwbwbw"
    precomputed = set()
    for i in range(12):
        for j in range(i, 12):
            substring = base[i:j+1]
            w = substring.count('w')
            b = substring.count('b')
            precomputed.add((w, b))
    max_m = min(W // 7, B // 5)
    found = False
    for m in range(0, max_m + 1):
        a = W - 7 * m
        b_val = B - 5 * m
        if a < 0 or b_val < 0:
            continue
        if (a, b_val) in precomputed:
            print("Yes")
            exit()
    print("No")