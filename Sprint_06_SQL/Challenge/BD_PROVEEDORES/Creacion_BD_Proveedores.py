import pandas as pd
import sqlite3

# Conectar a la base de datos
connection = sqlite3.connect("ProveedoresV2.db")
cursor_gestion = connection.cursor()

# Crear las tablas (si no existen)
query_create = '''
CREATE TABLE IF NOT EXISTS Proveedores (
    ProveedorID INT PRIMARY KEY,
    Nombre VARCHAR(255),
    Dirección VARCHAR(255),
    Ciudad VARCHAR(255),
    Provincia VARCHAR(255),
    CodigoProveedor VARCHAR(50) UNIQUE
);

CREATE TABLE IF NOT EXISTS Categorias (
    CategoriaID INT PRIMARY KEY,
    Nombre VARCHAR(255),
    CodigoCategoria VARCHAR(50) UNIQUE
);

CREATE TABLE IF NOT EXISTS Piezas (
    PiezaID INT PRIMARY KEY,
    Nombre VARCHAR(255),
    Color VARCHAR(50),
    Precio DECIMAL(10, 2),
    CategoriaID INT,
    FOREIGN KEY (CategoriaID) REFERENCES Categorias(CategoriaID)
);

CREATE TABLE IF NOT EXISTS Suministros (
    SuministroID INT PRIMARY KEY,
    ProveedorID INT,
    PiezaID INT,
    Cantidad INT,
    Fecha DATE,
    FOREIGN KEY (ProveedorID) REFERENCES Proveedores(ProveedorID),
    FOREIGN KEY (PiezaID) REFERENCES Piezas(PiezaID)
);
'''
cursor_gestion.executescript(query_create)

# Guardar los cambios y cerrar la conexión
connection.commit()

# Imprimir la ruta completa de la base de datos
print("Base de datos creada exitosamente en:", connection.execute('PRAGMA database_list;').fetchall()[0][2])

# Cerrar la conexión
connection.close()
