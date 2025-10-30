def can_form_abc(s):
    if s == "abc":
        return "YES"
    if s == "acb":
        return "YES"
    if s == "bac":
        return "YES"
    if s == "bca":
        return "NO"
    if s == "cab":
        return "NO"
    if s == "cba":
        return "YES"

t = int(input())
for _ in range(t):
    s = input().strip()
    print(can_form_abc(s))