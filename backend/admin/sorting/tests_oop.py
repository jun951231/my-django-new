import unittest

from models_oop import Calculator, Contacts, Grade

class TestCalculator(unittest.TestCase):

    def test_add(self):
        instance = Calculator(5, 8)
        res = instance.add()
        self.assertEqual(res, 13)

    def test_substract(self):
        instance = Calculator(9, 2)
        res = instance.substract()
        self.assertEqual(res, 7)

    def test_multiple(self):
        instance = Calculator(5, 5)
        res = instance.multiple()
        self.assertEqual(res, 25)

    def test_divide(self):
        instance = Calculator(18, 3)
        res = instance.divide()
        self.assertEqual(res, 6)

class TestContacts(unittest.TestCase):

    def test_get_contact(self):
        ls = []
        ls.append(Contacts.set_contact('Jimmy', '010-1231-1231', 'jim@test.com', 'New York'))
        ls.append(Contacts.set_contact('Timmy', '010-1232-1232', 'tim@test.com', 'Wasinton'))
        ls.append(Contacts.set_contact('Licky', '010-1233-1233', 'lick@test.com', 'LA'))
        ls = Contacts.get_contact(ls)
        self.assertEqual(ls[0].name, 'Jimmy')

    def test_del_contact(self):
        ls = []
        ls.append(Contacts.set_contact('Jimmy', '010-1231-1231', 'jim@test.com', 'New York'))
        ls.append(Contacts.set_contact('Timmy', '010-1232-1232', 'tim@test.com', 'Wasinton'))
        ls.append(Contacts.set_contact('Licky', '010-1233-1233', 'lick@test.com', 'LA'))
        ls = Contacts.del_contact(ls, 'Jimmy')
        print([x.to_string() for x in ls])
        self.assertEqual(len(ls), 2)

class TestGrade(unittest.TestCase):

    def test_grade(self):
        grade = Grade()
        grade._kor = 60
        grade._eng = 80
        grade._math = 70
        self.assertEqual(grade.return_grade(), 'C')


if __name__ == '__main__':
    unittest.main()