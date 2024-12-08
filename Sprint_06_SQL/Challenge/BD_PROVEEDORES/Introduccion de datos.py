import sqlite3

# Conectar a la base de datos
connection = sqlite3.connect("ProveedoresV2.db")
cursor_gestion = connection.cursor()

def insertar_proveedor():
    print("Introducir datos del proveedor:")
    proveedor_id = int(input("Proveedor ID: "))
    nombre = input("Nombre: ")
    direccion = input("Dirección: ")
    ciudad = input("Ciudad: ")
    provincia = input("Provincia: ")
    codigo_proveedor = input("Código de Proveedor: ")
    
    cursor_gestion.execute("INSERT INTO Proveedores VALUES (?, ?, ?, ?, ?, ?)",
                           (proveedor_id, nombre, direccion, ciudad, provincia, codigo_proveedor))
    connection.commit()
    print("Proveedor añadido correctamente.")
    visualizar_proveedores()

def insertar_categoria():
    print("Introducir datos de la categoría:")
    categoria_id = int(input("Categoría ID: "))
    nombre = input("Nombre: ")
    codigo_categoria = input("Código de Categoría: ")
    
    cursor_gestion.execute("INSERT INTO Categorias VALUES (?, ?, ?)",
                           (categoria_id, nombre, codigo_categoria))
    connection.commit()
    print("Categoría añadida correctamente.\n")

def insertar_pieza():
    print("Introducir datos de la pieza:")
    pieza_id = int(input("Pieza ID: "))
    nombre = input("Nombre: ")
    color = input("Color: ")
    precio = float(input("Precio: "))
    categoria_id = int(input("Categoría ID: "))
    
    cursor_gestion.execute("INSERT INTO Piezas VALUES (?, ?, ?, ?, ?)",
                           (pieza_id, nombre, color, precio, categoria_id))
    connection.commit()
    print("Pieza añadida correctamente.\n")

def insertar_suministro():
    print("Introducir datos del suministro:")
    suministro_id = int(input("Suministro ID: "))
    proveedor_id = int(input("Proveedor ID: "))
    pieza_id = int(input("Pieza ID: "))
    cantidad = int(input("Cantidad: "))
    fecha = input("Fecha (YYYY-MM-DD): ")
    
    cursor_gestion.execute("INSERT INTO Suministros VALUES (?, ?, ?, ?, ?)",
                           (suministro_id, proveedor_id, pieza_id, cantidad, fecha))
    connection.commit()
    print("Suministro añadido correctamente.\n")

def visualizar_proveedores():
    print("Datos de la tabla Proveedores:")
    cursor_gestion.execute("SELECT * FROM Proveedores")
    filas = cursor_gestion.fetchall()
    for fila in filas:
        print(fila)
    print("\n")

def visualizar_tabla(nombre_tabla):
    print(f"Datos de la tabla {nombre_tabla}:")
    cursor_gestion.execute(f"SELECT * FROM {nombre_tabla}")
    filas = cursor_gestion.fetchall()
    for fila in filas:
        print(fila)
    print("\n")

def menu():
    while True:
        print("Seleccione una opción:")
        print("1. Añadir Proveedor")
        print("2. Añadir Categoría")
        print("3. Añadir Pieza")
        print("4. Añadir Suministro")
        print("5. Visualizar Proveedores")
        print("6. Visualizar Categorías")
        print("7. Visualizar Piezas")
        print("8. Visualizar Suministros")
        print("9. Salir")
        opcion = input("Opción: ")
        
        if opcion == "1":
            insertar_proveedor()
        elif opcion == "2":
            insertar_categoria()
        elif opcion == "3":
            insertar_pieza()
        elif opcion == "4":
            insertar_suministro()
        elif opcion == "5":
            visualizar_tabla("Proveedores")
        elif opcion == "6":
            visualizar_tabla("Categorias")
        elif opcion == "7":
            visualizar_tabla("Piezas")
        elif opcion == "8":
            visualizar_tabla("Suministros")
        elif opcion == "9":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Intente nuevamente.")

menu()

# Cerrar la conexión
connection.close()



