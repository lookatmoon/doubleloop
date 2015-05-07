
class MyPool:
	def __init__(self, data):
		self.data = data

class MyClass:
	def __init__(self, item):
		self.item = item

	def print_item(self):
		print self.item.data

	def modify_item(self):
		self.item.data['c'] = 3

item = {'a':1}
pool = MyPool(item)

obj = MyClass(pool)
obj.print_item()

pool.data.update({'b':2})
obj.print_item()

obj.modify_item()
obj.print_item()
