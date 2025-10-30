base = 'wbwbwwbwbwbw'
s_long = base * 20  # Generates a sufficiently long string to cover all possible cases

W, B = map(int, input().split())
required_length = W + B

found = False
if required_length != 0:
    max_start = len(s_long) - required_length
    if max_start >= 0:
        for i in range(max_start + 1):
            substring = s_long[i:i+required_length]
            cnt_w = substring.count('w')
            cnt_b = substring.count('b')
            if cnt_w == W and cnt_b == B:
                found = True
                break

print("Yes" if found else "No")