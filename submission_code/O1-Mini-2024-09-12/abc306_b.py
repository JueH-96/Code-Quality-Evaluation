bits = list(map(int, input().split()))
result = sum(bit * (1 << i) for i, bit in enumerate(bits))
print(result)