pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
n = int(input())

decimal_index = pi_str.find('.')
integer_part = pi_str[:decimal_index]
decimal_part = pi_str[decimal_index+1:]

truncated_decimal_part = decimal_part[:n]

result = integer_part + "." + truncated_decimal_part
print(result)