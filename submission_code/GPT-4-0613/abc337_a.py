N = int(input())
Takahashi = 0
Aoki = 0
for _ in range(N):
    x, y = map(int, input().split())
    Takahashi += x
    Aoki += y
if Takahashi > Aoki:
    print('Takahashi')
elif Takahashi < Aoki:
    print('Aoki')
else:
    print('Draw')