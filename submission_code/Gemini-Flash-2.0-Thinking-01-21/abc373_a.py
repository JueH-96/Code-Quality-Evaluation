string_list = []
for _ in range(12):
    s = input()
    string_list.append(s)

count = 0
for i in range(12):
    if len(string_list[i]) == (i + 1):
        count += 1

print(count)