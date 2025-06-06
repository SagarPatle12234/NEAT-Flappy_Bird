import pygame
import random
import os

pygame.init()

# Game constants
WIN_WIDTH = 500
WIN_HEIGHT = 800
FLOOR_HEIGHT = 70
FLOOR = WIN_HEIGHT - FLOOR_HEIGHT
PIPE_GAP = 200
VEL = 5
GRAVITY = 1.5

# Load and scale assets
BIRD_IMGS = [
    pygame.transform.scale2x(pygame.image.load("bird1.png")),
    pygame.transform.scale2x(pygame.image.load(("bird2.png")),
    pygame.transform.scale2x(pygame.image.load("bird3.png"))
]

PIPE_IMG = pygame.transform.scale2x(pygame.image.load( "pipe.png"))

BG_IMG = pygame.transform.scale(
    pygame.image.load("bg.png"),
    (WIN_WIDTH, WIN_HEIGHT)
)

BASE_IMG = pygame.transform.scale(
    pygame.image.load("base.png"),
    (WIN_WIDTH, FLOOR_HEIGHT)
)

STAT_FONT = pygame.font.SysFont("comicsans", 50)

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")


# Classes
class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -7.5  # Modified jump to be lower
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        displacement = self.vel * self.tick_count + 0.5 * GRAVITY * self.tick_count ** 2

        if displacement >= 10:
            displacement = 10
        if displacement < 0:
            displacement -= 2

        self.y += displacement

        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1
        self.img = self.IMGS[self.img_count // self.ANIMATION_TIME % 3]

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)

        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = PIPE_GAP
    VEL = VEL

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        return b_point or t_point


class Base:
    VEL = VEL
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


# Utility function to draw everything
def draw_window(win, bird, pipes, base, score):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    bird.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    pygame.display.update()


# Only runs if you run main.py directly
if __name__ == "__main__":
    def main():
        bird = Bird(230, 350)
        base = Base(FLOOR)
        pipes = [Pipe(600)]
        score = 0

        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                bird.jump()

            bird.move()
            base.move()

            add_pipe = False
            rem = []
            for pipe in pipes:
                pipe.move()
                if pipe.collide(bird):
                    print("Hit!")
                    run = False

                if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                    rem.append(pipe)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if add_pipe:
                score += 1
                pipes.append(Pipe(600))

            for r in rem:
                pipes.remove(r)

            if bird.y + bird.img.get_height() >= FLOOR or bird.y < 0:
                print("Fell!")
                run = False

            draw_window(WIN, bird, pipes, base, score)

    main()
