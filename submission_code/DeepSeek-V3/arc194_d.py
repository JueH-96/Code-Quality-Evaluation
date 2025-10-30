n = int(input())
A = list(map(int, input().split()))
total = sum(A)
max_a = max(A)
if max_a > total - max_a:
    print("Fennec" if (total - (total - max_a) * 2) % 2 == 1 else "Snuke")
else:
    print("Fennec" if total % 2 == 1 else "Snuke")