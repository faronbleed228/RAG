from sys import stderr

if __name__ == "__main__":
    try:
        from .main import main
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("Операция прервана.", file=stderr)
        exit(130)
    except Exception as error:
        print(f"Ошибка: {error}", file=stderr)
        exit(1)
