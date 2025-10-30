N = int(input())
S = input()

pos = S.find("ABC")
if pos == -1:
    print(-1)
else:
    print(pos + 1)  # Convert from 0-indexed to 1-indexed