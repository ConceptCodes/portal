from typer.testing import CliRunner

from portal.cli import app

runner = CliRunner()


class TestCLI:
    def test_help_succeeds(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Portal" in result.stdout

    def test_process_help(self) -> None:
        result = runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0
        assert "INPUT" in result.stdout
        assert "OUTPUT" in result.stdout

    def test_live_help(self) -> None:
        result = runner.invoke(app, ["live", "--help"])
        assert result.exit_code == 0
        assert "CAMERA_ID" in result.stdout

    def test_process_no_args_fails(self) -> None:
        result = runner.invoke(app, ["process"])
        assert result.exit_code != 0

    def test_process_missing_input(self) -> None:
        result = runner.invoke(app, ["process", "nonexistent.mp4", "out.mp4"])
        assert result.exit_code != 0

    def test_build_config_defaults(self) -> None:
        from portal.cli import _build_config

        config = _build_config(
            model="yolov8n.pt",
            conf=0.5,
            padding=0.1,
            width=1280,
            height=720,
            alpha=0.1,
            jump_threshold=0.15,
            track_id=None,
            show=False,
            skip=0,
        )
        assert config.model == "yolov8n.pt"

    def test_process_with_all_options(self) -> None:
        result = runner.invoke(
            app,
            [
                "process",
                "--model",
                "yolov8n.pt",
                "--conf",
                "0.7",
                "--padding",
                "0.2",
                "--width",
                "1920",
                "--height",
                "1080",
                "--alpha",
                "0.05",
                "--jump-threshold",
                "0.2",
                "--track-id",
                "5",
                "--show",
                "--skip",
                "2",
                "nonexistent.mp4",
                "out.mp4",
            ],
        )
        assert result.exit_code != 0
