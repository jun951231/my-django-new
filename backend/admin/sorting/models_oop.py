class Calculator(object):
    num1: int
    num2: int

    @property
    def num1(self):
        return self._num1

    @num1.setter
    def num1(self, num1):
        self._num1 = num1

    @property
    def num2(self):
        return self._num2

    @num2.setter
    def num2(self, num2):
        self._num2 = num2

    def add(self):
        return self._num1 + self._num2

    def substract(self):
        return self._num1 - self._num2

    def multiple(self):
        return self._num1 * self._num2

    def divide(self):
        return self._num1 / self._num2

class Contacts(object):
    def __init__(self, name, phone, email, address):
        self.name = name
        self.phone = phone
        self.email = email
        self.address = address

    def to_string(self):
        print(f'{self.name}, {self.phone}, {self.email}, {self.address}')

    @staticmethod
    def set_contact(name, phone, email, address) -> object:
        return Contacts(name, phone, email, address)

    @staticmethod
    def get_contact(ls):
        for i in ls:
            i.to_string()
        return ls

    @staticmethod
    def del_contact(ls, name):
        for i, j in enumerate(ls):
            if name == j.name:
                del ls[i]
        return ls

class Grade(object):
    kor: int
    eng: int
    math: int

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def kor(self):
        return self._kor

    @kor.setter
    def kor(self, kor):
        self._kor = kor

    @property
    def eng(self):
        return self._eng

    @eng.setter
    def eng(self, eng):
        self._eng = eng

    @property
    def math(self):
        return self._math

    @math.setter
    def math(self, math):
        self._math = math

    def sum(self):
        return self._kor + self._eng + self._math

    def avg(self):
        return float(self.sum() / 3)

    def return_grade(self) -> str:
        aver = self.avg()
        if aver >= 90:
            return 'A'
        elif aver >= 80:
            return 'B'
        elif aver >= 70:
            return 'C'
        elif aver >= 60:
            return 'D'
        elif aver >= 50:
            return 'E'
        else:
            return 'F'

