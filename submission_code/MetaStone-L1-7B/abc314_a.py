n = int(input())
pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
parts = pi_str.split('.')
integer_part = parts[0]
decimal_part = parts[1][:n]
result = f"{integer_part}.{decimal_part}"
print(result)