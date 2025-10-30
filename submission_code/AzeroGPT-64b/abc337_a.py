n=int(input())
a=0
b=0
for i in range(n):
  x,y=list(map(int,input().split()))
  a+=x
  b+=y
if a==b:
  print('Draw')
elif a>b:
  print('Takahashi')
else:
  print('Aoki')