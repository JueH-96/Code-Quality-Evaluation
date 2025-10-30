n, k = map(int, input().split())
A = list(map(int, input().split()))

# lista de comprensión para dividir los múltiplos de k por k
B = [a//k for a in A if a % k == 0]

# imprimir los elementos de la lista separados por un espacio
print(*sorted(B))