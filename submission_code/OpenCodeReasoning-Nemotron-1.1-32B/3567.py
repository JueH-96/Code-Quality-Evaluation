class Solution:
	def convertDateToBinary(self, date: str) -> str:
		parts = date.split('-')
		year_bin = bin(int(parts[0]))[2:]
		month_bin = bin(int(parts[1]))[2:]
		day_bin = bin(int(parts[2]))[2:]
		return f"{year_bin}-{month_bin}-{day_bin}"