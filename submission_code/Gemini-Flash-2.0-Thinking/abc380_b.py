s = input()
parts = s.split('|')
hyphen_groups = parts[1:-1]
a = [len(group) for group in hyphen_groups]
print(*a)