S = input().strip()
even_positions = S[1:16:2]
if all(c == '0' for c in even_positions):
    print("Yes")
else:
    print("No")