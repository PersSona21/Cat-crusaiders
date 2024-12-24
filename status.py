import krpc
conn = krpc.connect(name='Проверка соединения')
print(conn.krpc.get_status().version)
print('Соединение установлено\nРакета готова к запуску')