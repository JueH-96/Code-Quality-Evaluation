S = input()
T = ""
for char in S:
    if char not in "aeiou":
        T += char
print(T)