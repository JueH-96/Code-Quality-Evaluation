S = input()

def remove_abc(s: str) -> str:
    while "ABC" in s:
        index = s.index("ABC")
        s = s[:index] + s[index + 3:]
    return s


print(remove_abc(S))