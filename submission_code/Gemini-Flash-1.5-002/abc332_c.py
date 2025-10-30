# YOUR CODE HERE
n, m = map(int, input().split())
s = input()

logo_count = 0
plain_count = 0
logo_used = [False] * 1000
plain_used = [False] * 1000
ans = 0

for i in range(n):
    if s[i] == '0':
        logo_count = 0
        plain_count = 0
        for j in range(1000):
            logo_used[j] = False
            plain_used[j] = False
    elif s[i] == '1':
        found = False
        for j in range(m):
            if not plain_used[j]:
                plain_used[j] = True
                found = True
                break
        if not found:
            found_logo = False
            for j in range(1000):
                if not logo_used[j]:
                    logo_used[j] = True
                    found_logo = True
                    break
            if not found_logo:
                ans += 1
                logo_used[ans-1] = True
    elif s[i] == '2':
        found = False
        for j in range(1000):
            if not logo_used[j]:
                logo_used[j] = True
                found = True
                break
        if not found:
            ans += 1
            logo_used[ans-1] = True

print(ans)