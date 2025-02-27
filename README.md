## Лабораторная работа 1. «Решение системы линейных алгебраических уравнений СЛАУ»
### Вариант 3 - Метод Гаусса-Зейделя.

1. Номер варианта определяется как номер в списке группы согласно ИСУ.
2. В программе численный метод должен быть реализован в виде отдель-
ной подпрограммы/метода/класса, в который исходные/выходные дан-
ные передаются в качестве параметров.
3. Размерность матрицы n<=20 (задается из файла или с клавиатуры - по
выбору конечного пользователя).
4. Должна быть реализована возможность ввода коэффициентов матрицы,
как с клавиатуры, так и из файла (по выбору конечного пользователя).
Для прямых методов должно быть реализовано:

Для итерационных методов должно быть реализовано:
- Точность задается с клавиатуры/файла,
- Проверка диагонального преобладания (в случае, если диагональное
преобладание в исходной матрице отсутствует, сделать перестановку
строк/столбцов до тех пор, пока преобладание не будет достигнуто). В
случае невозможности достижения диагонального преобладания -
выводить соответствующее сообщение.
- Вывод нормы матрицы (любой, на Ваш выбор),
- Вывод вектора неизвестных: x_1, x_2, ..., x_n,
- Вывод количества итераций, за которое было найдено решение,
- Вывод вектора погрешностей: |x_i^{(k)} - x_i^{(k-1)}|.

Содержание отчета:
- Цель работы,
- Описание метода, расчетные формулы,
- Листинг программы (по крайне мере, где реализован сам метод)
- Примеры и результаты работы программы,
- Выводы.
- Отчет предоставляется в электронном/бумажном виде.
