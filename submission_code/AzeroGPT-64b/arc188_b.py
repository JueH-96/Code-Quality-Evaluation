for _ in range(int(input())):
  N, K = map(int, input().split())
  print('Yes' if 2 * K % N == 0 else 'No')