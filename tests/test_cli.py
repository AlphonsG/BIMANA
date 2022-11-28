import subprocess


def test_cli() -> None:
    assert 'help' in subprocess.check_output(['bimana', '-h'], text=True)
